import ActivityCard from './ActivityCard';

interface ActivitiesGridProps {
  onActivityClick: (id: number) => void;
}

export default function ActivitiesGrid({ onActivityClick }: ActivitiesGridProps) {
  const activities = [
    { id: 1, name: 'Activity - 1' },
    { id: 2, name: 'Activity - 2' },
    { id: 3, name: 'Activity - 3' },
    { id: 4, name: 'Activity - 4' }
  ];

  return (
    <div className="p-6">
      <div className="max-w-4xl mx-auto">
        <div className="grid grid-cols-2 gap-8 max-w-2xl mx-auto">
          {activities.map((activity) => (
            <ActivityCard
              key={activity.id}
              id={activity.id}
              name={activity.name}
              onClick={onActivityClick}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
